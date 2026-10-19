#pragma once

// Generated from the canonical scalar-flux shader by generate_weno_common_msl.py.
#define NIMBUS_WENO_COMMON_MSL \
"inline float weno_difference(float value, float center, float scale)\n" \
"{\n" \
"    float delta = value - center;\n" \
"    return (!isnan(delta) && !isinf(delta)) ? delta / scale : value / scale - center / scale;\n" \
"}\n" \
"\n" \
"inline float weno5_offset(float a, float b, float c, float d, float e)\n" \
"{\n" \
"    float magnitude = max(max(max(abs(a), abs(b)), max(abs(c), abs(d))), abs(e));\n" \
"    float scale = max(magnitude, 1e-20f);\n" \
"    if (isnan(scale) || isinf(scale))\n" \
"        return 0.0f;\n" \
"    float ac = weno_difference(a, c, scale), bc = weno_difference(b, c, scale);\n" \
"    float dc = weno_difference(d, c, scale), ec = weno_difference(e, c, scale);\n" \
"    float p0 = (1.0f / 3.0f) * ac - (7.0f / 6.0f) * bc;\n" \
"    float p1 = -(1.0f / 6.0f) * bc + (1.0f / 3.0f) * dc;\n" \
"    float p2 = (5.0f / 6.0f) * dc - (1.0f / 6.0f) * ec;\n" \
"    float s0a = ac - 2.0f * bc, s0b = ac - 4.0f * bc;\n" \
"    float s1a = bc + dc, s1b = bc - dc;\n" \
"    float s2a = ec - 2.0f * dc, s2b = ec - 4.0f * dc;\n" \
"    float b0 = (13.0f / 12.0f) * s0a * s0a + .25f * s0b * s0b;\n" \
"    float b1 = (13.0f / 12.0f) * s1a * s1a + .25f * s1b * s1b;\n" \
"    float b2 = (13.0f / 12.0f) * s2a * s2a + .25f * s2b * s2b;\n" \
"    float relative = magnitude / scale;\n" \
"    float floor_scale = 1e-20f / scale;\n" \
"    float eps = 1e-6f * relative * relative + floor_scale * floor_scale;\n" \
"    float minimum = min(eps + b0, min(eps + b1, eps + b2));\n" \
"    float r0 = minimum / (eps + b0), r1 = minimum / (eps + b1), r2 = minimum / (eps + b2);\n" \
"    float a0 = .1f * r0 * r0, a1 = .6f * r1 * r1, a2 = .3f * r2 * r2;\n" \
"    float offset = scale * ((a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2));\n" \
"    return (!isnan(offset) && !isinf(offset)) ? offset : 0.0f;\n" \
"}\n"
