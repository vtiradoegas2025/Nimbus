// Radiative faces must preserve gradual underflow even when the device has
// no FP32 denormal-preserve mode. Integer arithmetic rounds each operation
// to binary32, matching the host boundary contract without FP64 hardware.
uint boundary_shift_jam(uint value, uint distance)
{
    if (distance == 0u) return value;
    if (distance >= 32u) return uint(value != 0u);
    return (value >> distance) | uint((value << (32u - distance)) != 0u);
}

float boundary_add(float left, float right)
{
    uint a = floatBitsToUint(left), b = floatBitsToUint(right);
    uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    if (am >= 0x7f800000u || bm >= 0x7f800000u) return left + right;
    if (am < bm) { uint t = a; a = b; b = t; t = am; am = bm; bm = t; }
    uint sign = a & 0x80000000u;
    uint exponent = max(am >> 23, 1u);
    uint other_exponent = max(bm >> 23, 1u);
    uint x = ((am & 0x7fffffu) | (am >= 0x800000u ? 0x800000u : 0u)) << 3;
    uint y = ((bm & 0x7fffffu) | (bm >= 0x800000u ? 0x800000u : 0u)) << 3;
    y = boundary_shift_jam(y, exponent - other_exponent);
    if (((a ^ b) & 0x80000000u) == 0u)
    {
        x += y;
        if (x >= 0x8000000u) { x = boundary_shift_jam(x, 1u); ++exponent; }
    }
    else
    {
        x -= y;
        if (x == 0u) return uintBitsToFloat(0u);
        while (x < 0x4000000u && exponent > 1u) { x <<= 1; --exponent; }
    }
    uint rounded = x >> 3, remainder = x & 7u;
    if (remainder > 4u || (remainder == 4u && (rounded & 1u) != 0u)) ++rounded;
    if (rounded >= 0x1000000u) { rounded >>= 1; ++exponent; }
    if (exponent >= 255u) return uintBitsToFloat(sign | 0x7f800000u);
    if (rounded < 0x800000u) exponent = 0u;
    return uintBitsToFloat(sign | (exponent << 23) | (rounded & 0x7fffffu));
}

float boundary_sub(float left, float right)
{
    return boundary_add(left, uintBitsToFloat(floatBitsToUint(right) ^ 0x80000000u));
}

float boundary_mul(float left, float right)
{
    uint a = floatBitsToUint(left), b = floatBitsToUint(right);
    uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    uint sign = (a ^ b) & 0x80000000u;
    if (am >= 0x7f800000u || bm >= 0x7f800000u) return left * right;
    if (am == 0u || bm == 0u) return uintBitsToFloat(sign);
    int ae = int(max(am >> 23, 1u)), be = int(max(bm >> 23, 1u));
    uint x = (am & 0x7fffffu) | (am >= 0x800000u ? 0x800000u : 0u);
    uint y = (bm & 0x7fffffu) | (bm >= 0x800000u ? 0x800000u : 0u);
    while (x < 0x800000u) { x <<= 1; --ae; }
    while (y < 0x800000u) { y <<= 1; --be; }
    uint high, low;
    umulExtended(x, y, high, low);
    bool carry = (high & 0x8000u) != 0u;
    int exponent = ae + be - 127 + int(carry);
    int shift = 23 + int(carry) + max(1 - exponent, 0);
    if (shift > 48) return uintBitsToFloat(sign);
    uint rounded, remainder, halfway;
    bool sticky = false;
    if (shift < 32)
    {
        rounded = (low >> shift) | (high << (32 - shift));
        remainder = low & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1);
    }
    else if (shift == 32)
    {
        rounded = high; remainder = low; halfway = 0x80000000u;
    }
    else
    {
        rounded = high >> (shift - 32);
        remainder = high & ((1u << (shift - 32)) - 1u);
        halfway = 1u << (shift - 33);
        sticky = low != 0u;
    }
    if (remainder > halfway || (remainder == halfway && (sticky || (rounded & 1u) != 0u))) ++rounded;
    exponent = max(exponent, 1);
    if (rounded >= 0x1000000u) { rounded >>= 1; ++exponent; }
    if (exponent >= 255) return uintBitsToFloat(sign | 0x7f800000u);
    if (rounded < 0x800000u) exponent = 0;
    return uintBitsToFloat(sign | (uint(exponent) << 23) | (rounded & 0x7fffffu));
}
