float round_storage_units(vec2 units,uint sign)
{
    float whole = floor(units.x);
    float tail  = fma(-1.0, whole, units.x) + units.y;
    if (tail < 0)
    {
        whole -= 1;
        tail += 1;
    }
    if (tail >= 1)
    {
        whole += 1;
        tail -= 1;
    }
    if (tail > 0.5 || (tail == 0.5 && (uint(whole) & 1u) != 0u))
        whole += 1;
    return uintBitsToFloat(sign | uint(whole));
}
// Scale tiny inventories before arithmetic on devices that flush FP32 subnormals.
float scale_component(float value)
{
    uint bits      = floatBitsToUint(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude < 0x00800000u)
    {
        float scaled = float(magnitude) * 1.1102230246251565e-16;
        return (bits & 0x80000000u) != 0u ? -scaled : scaled;
    }
    return value * 7.922816251426434e28;
}
vec2 unscale(vec2 value, bool scaled)
{
    if (!scaled || any(isnan(value)) || any(isinf(value)))
        return value;
    if (abs(value.x) >= 9.313225746154785e-10)
        return df_mul(value, vec2(1.262177448353619e-29, 0));
    uint  sign  = floatBitsToUint(value.x) & 0x80000000u;
    vec2  units = df_mul(sign == 0u ? value : -value, vec2(9007199254740992.0, 0));
    return vec2(round_storage_units(units,sign),0);
}
