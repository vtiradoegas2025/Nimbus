uint pd_shift_jam(uint value, int distance)
{
    if (distance <= 0)
        return value;
    if (distance >= 32)
        return value == 0u ? 0u : 1u;
    return (value >> uint(distance)) | ((value << uint(32 - distance)) == 0u ? 0u : 1u);
}
// Guard, round and sticky bits survive alignment without denormal arithmetic.
uint pd_add_bits(uint a, uint b)
{
    uint ap = a & 0x7fffffffu, bp = b & 0x7fffffffu;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
        return 0x7fc00000u;
    if (ap == 0u)
        return b;
    if (bp == 0u)
        return a;
    if (ap < bp)
    {
        uint t = a;
        a      = b;
        b      = t;
        t      = ap;
        ap     = bp;
        bp     = t;
    }
    uint sign = a & 0x80000000u;
    int  ae = max(1, int(ap >> 23)), be = max(1, int(bp >> 23));
    uint am   = (ap & 0x007fffffu) | (ap >= 0x00800000u ? 0x00800000u : 0u);
    uint bm   = (bp & 0x007fffffu) | (bp >= 0x00800000u ? 0x00800000u : 0u);
    uint left = am << 3, right = pd_shift_jam(bm << 3, ae - be);
    uint total = ((a ^ b) & 0x80000000u) == 0u ? left + right : left - right;
    if (total == 0u)
        return 0u;
    if (total >= 0x08000000u)
    {
        total = pd_shift_jam(total, 1);
        ae += 1;
    }
    for (int i = 0; i < 26 && total < 0x04000000u && ae > 1; ++i)
    {
        total <<= 1;
        ae -= 1;
    }
    uint remainder = total & 7u;
    uint rounded   = total >> 3;
    bool increment = remainder > 4u || (remainder == 4u && (rounded & 1u) != 0u);
    if (increment)
        rounded += 1u;
    if (rounded >= 0x01000000u)
    {
        rounded >>= 1;
        ae += 1;
    }
    if (ae >= 255)
        return sign | 0x7f800000u;
    return sign | (ae == 1 && rounded < 0x00800000u ? rounded : (uint(ae) << 23) | (rounded & 0x007fffffu));
}
float pd_add(float a, float b)
{
    uint ab = floatBitsToUint(a), bb = floatBitsToUint(b);
    uint ap = ab & 0x7fffffffu, bp = bb & 0x7fffffffu;
    if (ap == 0u)
        return b;
    if (bp == 0u)
        return a;
    precise float result = a + b;
    uint          rp     = floatBitsToUint(result) & 0x7fffffffu;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
        return result;
    if (ap >= 0x00800000u && bp >= 0x00800000u && rp >= 0x00800000u)
        return result;
    return uintBitsToFloat(pd_add_bits(ab, bb));
}
float pd_normalized_mantissa(float value, out int exponent)
{
    uint bits = floatBitsToUint(value), fraction = bits & 0x007fffffu;
    int  encoded = int((bits >> 23) & 255u);
    if (encoded == 0)
    {
        int top  = findMSB(fraction);
        exponent = top - 148;
        fraction = (fraction << uint(23 - top)) & 0x007fffffu;
    }
    else
        exponent = encoded - 126;
    return uintBitsToFloat((bits & 0x80000000u) | 0x3f000000u | fraction);
}
// The exact 48-bit significand uses two uints; final packing is nearest-even.
uint pd_product_bits(uint ab, uint bb)
{
    uint ap = ab & 0x7fffffffu, bp = bb & 0x7fffffffu;
    uint sign = (ab ^ bb) & 0x80000000u;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
        return ap > 0x7f800000u || bp > 0x7f800000u || ap == 0u || bp == 0u ? 0x7fc00000u : sign | 0x7f800000u;
    if (ap == 0u || bp == 0u)
        return sign;
    int  ae, be;
    uint am      = (floatBitsToUint(pd_normalized_mantissa(uintBitsToFloat(ab), ae)) & 0x007fffffu) | 0x00800000u;
    uint bm      = (floatBitsToUint(pd_normalized_mantissa(uintBitsToFloat(bb), be)) & 0x007fffffu) | 0x00800000u;
    uint p0      = (am & 4095u) * (bm & 4095u);
    uint p1      = (am >> 12) * (bm & 4095u) + (am & 4095u) * (bm >> 12) + (p0 >> 12);
    uint p2      = (am >> 12) * (bm >> 12) + (p1 >> 12);
    uint low     = (p0 & 4095u) | ((p1 & 4095u) << 12) | (p2 << 24);
    uint high    = p2 >> 8;
    int  top     = high >= 0x8000u ? 47 : 46;
    int  encoded = ae + be + top + 79;
    if (encoded >= 255)
        return sign | 0x7f800000u;
    int shift = encoded > 0 ? top - 23 : -ae - be - 101;
    if (shift > 48)
        return sign | 0u;
    uint rounded;
    bool above, tie;
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
        int upper_shift = shift - 32;
        rounded         = upper_shift == 0 ? high : high >> uint(upper_shift);
        uint remainder  = upper_shift == 0 ? 0u : high & ((1u << uint(upper_shift)) - 1u);
        uint midpoint   = upper_shift == 0 ? 0u : 1u << uint(upper_shift - 1);
        above = upper_shift == 0 ? low > 0x80000000u : remainder > midpoint || (remainder == midpoint && low != 0u);
        tie   = upper_shift == 0 ? low == 0x80000000u : remainder == midpoint && low == 0u;
    }
    if (above || (tie && (rounded & 1u) != 0u))
        rounded += 1u;
    if (encoded <= 0)
        return sign | rounded;
    if (rounded >= 0x01000000u)
    {
        rounded >>= 1;
        encoded += 1;
    }
    return sign | (uint(encoded) << 23) | (rounded & 0x007fffffu);
}
float pd_product(float a, float b)
{
    uint          ab = floatBitsToUint(a), bb = floatBitsToUint(b);
    uint          ap = ab & 0x7fffffffu, bp = bb & 0x7fffffffu;
    precise float result = a * b;
    if (ap >= 0x00800000u && ap < 0x7f800000u && bp >= 0x00800000u && bp < 0x7f800000u &&
        (ap >> 23) + (bp >> 23) > 127u)
        return result;
    return uintBitsToFloat(pd_product_bits(ab, bb));
}
float pd_pack_scale(float normal, int scale)
{
    uint bits = floatBitsToUint(normal), sign = bits & 0x80000000u;
    int  encoded = int((bits >> 23) & 255u) + scale;
    if (encoded >= 255)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (encoded > 0)
        return uintBitsToFloat(sign | (uint(encoded) << 23) | (bits & 0x007fffffu));
    int shift = 1 - encoded;
    if (shift > 24)
        return uintBitsToFloat(sign);
    uint significand = 0x00800000u | (bits & 0x007fffffu);
    uint rounded = significand >> uint(shift), remainder = significand & ((1u << uint(shift)) - 1u);
    uint midpoint = 1u << uint(shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1u) != 0u))
        rounded += 1u;
    return uintBitsToFloat(sign | rounded);
}
float pd_positive_divide(float numerator, float denominator)
{
    uint n = floatBitsToUint(numerator), d = floatBitsToUint(denominator);
    if (d == 0u || d >= 0x7f800000u || n >= 0x7f800000u)
        return numerator / denominator;
    if ((n & 0x7fffffffu) == 0u)
        return 0.0;
    uint ne = n >> 23, de = d >> 23;
    if (ne != 0u && de != 0u && d <= 0x7e800000u && int(ne) - int(de) >= -125)
    {
        precise float result = numerator / denominator;
        return result;
    }
    int           n_exponent, d_exponent;
    float         nm    = pd_normalized_mantissa(numerator, n_exponent);
    float         dm    = pd_normalized_mantissa(denominator, d_exponent);
    precise float ratio = nm / dm;
    return pd_pack_scale(ratio, n_exponent - d_exponent);
}

float pd_inventory(float rho, float q)
{
    return pd_product(rho, q);
}
bool pd_positive(float value)
{
    uint bits = floatBitsToUint(value);
    return (bits & 0x80000000u) == 0u && (bits & 0x7fffffffu) != 0u;
}
float pd_outward(float value)
{
    return pd_positive(value) ? value : 0.0;
}
float pd_reverse(float value)
{
    return uintBitsToFloat(floatBitsToUint(value) ^ 0x80000000u);
}
float pd_scaled_face(float value, float factor)
{
    float result = pd_product(value, factor);
    return result;
}
float pd_rate(float xp, float xm, float yp, float ym, float zp, float zm, float inv_x, float inv_y, float inv_z)
{
    precise float txp  = pd_product(pd_outward(xp), inv_x);
    precise float txm  = pd_product(pd_outward(pd_reverse(xm)), inv_x);
    precise float typ  = pd_product(pd_outward(yp), inv_y);
    precise float tym  = pd_product(pd_outward(pd_reverse(ym)), inv_y);
    precise float tzp  = pd_product(pd_outward(zp), inv_z);
    precise float tzm  = pd_product(pd_outward(pd_reverse(zm)), inv_z);
    precise float rate = 0.0;
    rate               = pd_add(rate, txp);
    rate               = pd_add(rate, txm);
    rate               = pd_add(rate, typ);
    rate               = pd_add(rate, tym);
    rate               = pd_add(rate, tzp);
    rate               = pd_add(rate, tzm);
    return rate;
}
float pd_remaining(float rho, float q, float dt, float outgoing)
{
    float inventory = pd_product(rho, q);
    float removed   = pd_product(dt, outgoing);
    float remaining = pd_add(inventory, pd_reverse(removed));
    return remaining;
}
float pd_next_inventory(float remaining, float dt, float incoming)
{
    float added  = pd_product(dt, incoming);
    float result = pd_add(remaining, added);
    return result;
}
bool pd_product_protected(float a, float b)
{
    uint ap = floatBitsToUint(a) & 0x7fffffffu, bp = floatBitsToUint(b) & 0x7fffffffu;
    return ap < 0x00800000u || bp < 0x00800000u || (ap >> 23) + (bp >> 23) <= 127u;
}
float pd_upper_product(float a, float b)
{
    uint ap = floatBitsToUint(a) & 0x7fffffffu, bp = floatBitsToUint(b) & 0x7fffffffu;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
        return uintBitsToFloat(0x7f800000u);
    float value = pd_product(a, b);
    uint  bits  = floatBitsToUint(value) & 0x7fffffffu;
    return pd_product_protected(a, b) ? value : uintBitsToFloat(bits < 0x7f800000u ? bits + 1u : 0x7f800000u);
}
float pd_upper_sum(float a, float b)
{
    uint ap = floatBitsToUint(a) & 0x7fffffffu, bp = floatBitsToUint(b) & 0x7fffffffu;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
        return uintBitsToFloat(0x7f800000u);
    if (ap == 0u)
        return b;
    if (bp == 0u)
        return a;
    float value = pd_add(a, b);
    uint  bits  = floatBitsToUint(value) & 0x7fffffffu;
    return ap < 0x00800000u || bp < 0x00800000u ? value : uintBitsToFloat(bits < 0x7f800000u ? bits + 1u : 0x7f800000u);
}
float pd_lower_inventory(float a, float b)
{
    uint ap = floatBitsToUint(a) & 0x7fffffffu, bp = floatBitsToUint(b) & 0x7fffffffu;
    if (ap >= 0x7f800000u || bp >= 0x7f800000u)
        return uintBitsToFloat(0x7f800000u);
    float value = pd_product(a, b);
    uint  bits  = floatBitsToUint(value) & 0x7fffffffu;
    return pd_product_protected(a, b) ? value : uintBitsToFloat(bits < 0x7f800000u ? bits - 1u : 0x7f800000u);
}
// Limiter and update instructions may use different underflow modes.
bool pd_interval_fits(float r,
                      float rho,
                      float q,
                      float dt,
                      float xp,
                      float xm,
                      float yp,
                      float ym,
                      float zp,
                      float zm,
                      float inv_x,
                      float inv_y,
                      float inv_z)
{
    float lower     = pd_lower_inventory(rho, q);
    float rate      = 0.0;
    rate            = pd_upper_sum(rate, pd_upper_product(pd_upper_product(pd_outward(xp), r), inv_x));
    rate            = pd_upper_sum(rate, pd_upper_product(pd_upper_product(pd_outward(pd_reverse(xm)), r), inv_x));
    rate            = pd_upper_sum(rate, pd_upper_product(pd_upper_product(pd_outward(yp), r), inv_y));
    rate            = pd_upper_sum(rate, pd_upper_product(pd_upper_product(pd_outward(pd_reverse(ym)), r), inv_y));
    rate            = pd_upper_sum(rate, pd_upper_product(pd_upper_product(pd_outward(zp), r), inv_z));
    rate            = pd_upper_sum(rate, pd_upper_product(pd_upper_product(pd_outward(pd_reverse(zm)), r), inv_z));
    float upper     = pd_upper_product(dt, rate);
    uint  low_bits  = floatBitsToUint(lower);
    uint  high_bits = floatBitsToUint(upper);
    return low_bits < 0x7f800000u && high_bits < 0x7f800000u && low_bits >= high_bits;
}

float pd_verified_factor(float candidate,
                         float rho,
                         float q,
                         float dt,
                         float xp,
                         float xm,
                         float yp,
                         float ym,
                         float zp,
                         float zm,
                         float inv_x,
                         float inv_y,
                         float inv_z)
{
    uint high = floatBitsToUint(candidate);
    if (high > 0x3f800000u)
        return candidate;
    if (pd_interval_fits(candidate, rho, q, dt, xp, xm, yp, ym, zp, zm, inv_x, inv_y, inv_z))
        return candidate;
    if (!pd_interval_fits(0.0, rho, q, dt, xp, xm, yp, ym, zp, zm, inv_x, inv_y, inv_z))
        return candidate;
    if ((floatBitsToUint(pd_lower_inventory(rho, q)) & 0x7fffffffu) == 0u)
    {
        return 0.0;
    }
    uint low = 0u;
    for (int iteration = 0; iteration < 32 && low < high; ++iteration)
    {
        uint middle = low + (high - low + 1u) / 2u;
        if (pd_interval_fits(uintBitsToFloat(middle), rho, q, dt, xp, xm, yp, ym, zp, zm, inv_x, inv_y, inv_z))
            low = middle;
        else
            high = middle - 1u;
    }
    return uintBitsToFloat(low);
}
