// Enable software product compensation only where the kernel error contract requires it.
// Expanding it throughout monolithic physics kernels can exhaust driver compiler memory.
#ifdef NIMBUS_PORTABLE_COMPENSATED_PRODUCT
float df_normalized_mantissa(float value, out int exponent)
{
    uint bits     = floatBitsToUint(value);
    uint fraction = bits & 0x007fffffu;
    int  encoded  = int((bits >> 23) & 255u);
    if (encoded == 0)
    {
        int top  = findMSB(fraction);
        exponent = top - 148;
        fraction = (fraction << uint(23 - top)) & 0x007fffffu;
    }
    else
    {
        exponent = encoded - 126;
    }
    return uintBitsToFloat((bits & 0x80000000u) | 0x3f000000u | fraction);
}
// Devices may flush subnormal operands/products even when normal multiplication
// is correctly rounded. Build only those products from their exact 48-bit significand.
float df_product(float a, float b)
{
    precise float product = a * b;
    uint          ab      = floatBitsToUint(a);
    uint          bb      = floatBitsToUint(b);
    uint          ap      = ab & 0x7fffffffu;
    uint          bp      = bb & 0x7fffffffu;
    uint          pp      = floatBitsToUint(product) & 0x7fffffffu;
    uint sign = (ab ^ bb) & 0x80000000u;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
    {
        return ap > 0x7f800000u || bp > 0x7f800000u || ap == 0u || bp == 0u
            ? uintBitsToFloat(0x7fc00000u) : uintBitsToFloat(sign | 0x7f800000u);
    }
    if (ap >= 0x00800000u && bp >= 0x00800000u && pp >= 0x00800000u)
    {
        return product;
    }
    if (ap == 0u || bp == 0u)
    {
        return uintBitsToFloat(sign);
    }
    int  ae, be;
    uint am      = (floatBitsToUint(df_normalized_mantissa(a, ae)) & 0x007fffffu) | 0x00800000u;
    uint bm      = (floatBitsToUint(df_normalized_mantissa(b, be)) & 0x007fffffu) | 0x00800000u;
    uint p0      = (am & 4095u) * (bm & 4095u);
    uint p1      = (am >> 12) * (bm & 4095u) + (am & 4095u) * (bm >> 12) + (p0 >> 12);
    uint p2      = (am >> 12) * (bm >> 12) + (p1 >> 12);
    uint low     = (p0 & 4095u) | ((p1 & 4095u) << 12) | (p2 << 24);
    uint high    = p2 >> 8;
    int  top     = high >= 0x8000u ? 47 : 46;
    int  encoded = ae + be + top + 79;
    int  shift   = encoded > 0 ? top - 23 : -ae - be - 101;
    if (shift > 48)
    {
        return uintBitsToFloat(sign);
    }
    uint rounded;
    bool above;
    bool tie;
    if (shift < 32)
    {
        rounded        = (high << uint(32 - shift)) | (low >> uint(shift));
        uint remainder = low & ((1u << uint(shift)) - 1u);
        uint midpoint  = 1u << uint(shift - 1);
        above          = remainder > midpoint;
        tie            = remainder == midpoint;
    }
    else
    {
        int upperShift = shift - 32;
        rounded        = upperShift == 0 ? high : high >> uint(upperShift);
        uint remainder = upperShift == 0 ? 0u : high & ((1u << uint(upperShift)) - 1u);
        uint midpoint  = upperShift == 0 ? 0u : 1u << uint(upperShift - 1);
        above = upperShift == 0 ? low > 0x80000000u : remainder > midpoint || (remainder == midpoint && low != 0u);
        tie   = upperShift == 0 ? low == 0x80000000u : remainder == midpoint && low == 0u;
    }
    if (above || (tie && (rounded & 1u) != 0u))
        rounded += 1u;
    if (encoded <= 0)
        return uintBitsToFloat(sign | rounded);
    if (rounded >= 0x01000000u)
    {
        rounded >>= 1;
        encoded += 1;
    }
    return uintBitsToFloat(sign | (uint(encoded) << 23) | (rounded & 0x007fffffu));
}
float df_scale_product_error(float value, int exponent)
{
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    if ((bits & 0x7fffffffu) == 0u)
    {
        return 0.0;
    }
    int encoded = int((bits >> 23) & 255u) + exponent;
    if (encoded >= 255)
    {
        return uintBitsToFloat(sign | 0x7f800000u);
    }
    if (encoded > 0)
    {
        return uintBitsToFloat(sign | (uint(encoded) << 23) | (bits & 0x007fffffu));
    }
    int shift = 1 - encoded;
    if (shift > 24)
    {
        return uintBitsToFloat(sign);
    }
    uint significand = 0x00800000u | (bits & 0x007fffffu);
    uint rounded     = significand >> uint(shift);
    uint remainder   = significand & ((1u << uint(shift)) - 1u);
    uint midpoint    = 1u << uint(shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1u) != 0u))
    {
        rounded += 1u;
    }
    return uintBitsToFloat(sign | rounded);
}
// Vulkan permits GLSL.std.450 Fma to have separate multiply/add precision.
// Split normalized significands so a rounded Fma cannot erase the residual;
// rescale its bits with ties-to-even to retain subnormal results.
// https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation
float df_product_error(float a, float b, float product)
{
    if (isnan(product))
    {
        return product;
    }
    if (isinf(product))
    {
        return isinf(a) || isinf(b) ? uintBitsToFloat(0x7fc00000u) : -product;
    }
    if ((floatBitsToUint(product) & 0x7fffffffu) == 0u)
    {
        return (floatBitsToUint(a) & 0x7fffffffu) == 0u || (floatBitsToUint(b) & 0x7fffffffu) == 0u ? 0.0 : product;
    }
    int           ae, be, pe;
    precise float am       = df_normalized_mantissa(a, ae);
    precise float bm       = df_normalized_mantissa(b, be);
    precise float pm       = df_normalized_mantissa(product, pe);
    int           exponent = ae + be;
    precise float rounded  = ldexp(pm, pe - exponent);
    precise float ah       = uintBitsToFloat(floatBitsToUint(am) & 0xfffff000u);
    precise float bh       = uintBitsToFloat(floatBitsToUint(bm) & 0xfffff000u);
    precise float al       = am - ah;
    precise float bl       = bm - bh;
    precise float e1       = rounded - ah * bh;
    precise float e2       = e1 - al * bh;
    precise float e3       = e2 - ah * bl;
    precise float error    = al * bl - e3;
    return df_scale_product_error(error, exponent);
}
#endif
vec2 df_add(vec2 a, vec2 b)
{
    float sum  = fma(1.0, a.x, b.x);
    float v    = fma(-1.0, a.x, sum);
    float tail = fma(1.0, fma(-1.0, v, b.x), fma(-1.0, fma(-1.0, v, sum), a.x));
    tail       = fma(1.0, tail, a.y);
    tail       = fma(1.0, tail, b.y);
    float high = fma(1.0, sum, tail);
    return vec2(high, fma(-1.0, fma(-1.0, sum, high), tail));
}
vec2 df_sub(vec2 a, vec2 b)
{
    return df_add(a, -b);
}
vec2 df_mul(vec2 a, vec2 b)
{
#ifdef NIMBUS_PORTABLE_COMPENSATED_PRODUCT
    float high = df_product(a.x, b.x);
    float tail = df_product_error(a.x, b.x, high);
#else
    float high = fma(a.x, b.x, 0.0);
    float tail = fma(a.x, b.x, -high);
#endif
    tail       = fma(1.0, tail, fma(a.x, b.y, 0.0));
    tail       = fma(1.0, tail, fma(a.y, b.x, 0.0));
    tail       = fma(1.0, tail, fma(a.y, b.y, 0.0));
    return df_add(vec2(high, 0.0), vec2(tail, 0.0));
}
vec2 df_div(vec2 a, vec2 b)
{
    float quotient  = a.x / b.x;
    vec2  remainder = df_sub(a, df_mul(b, vec2(quotient, 0.0)));
    return df_add(vec2(quotient, 0.0), vec2(fma(1.0, remainder.x, remainder.y) / b.x, 0.0));
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
    int correction = 0;
    if (x.x > 0.0 && x.x < 1.1754943508222875e-38)
    {
        x          = df_mul(x, vec2(16777216, 0));
        correction = -24;
    }
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
                  df_mul(vec2(0.69314718246459961, -1.904654323148236e-09), vec2(float(exponent + correction), 0.0)));
}
vec2 df_sqrt(vec2 x)
{
    float root      = sqrt(x.x);
    vec2  remainder = df_sub(x, df_mul(vec2(root, 0.0), vec2(root, 0.0)));
    return df_add(vec2(root, 0.0), df_div(remainder, vec2(2.0 * root, 0.0)));
}
