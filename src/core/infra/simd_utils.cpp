/**
 * @file simd_utils.cpp
 * @brief SIMD-accelerated vector math primitives.
 *
 * Provides architecture-aware implementations for element-wise arithmetic,
 * scalar-broadcast operations, clamping, sanitization, and reductions.
 * Each function dispatches to the best available ISA at compile time:
 * AVX (8 floats/iter), SSE (4), NEON (4), or scalar fallback (1).
 */

#include "util/simd_utils.hpp"

#include <cmath>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef SIMD_AVX512_AVAILABLE
        #include <immintrin.h>
    #elif defined(SIMD_AVX_AVAILABLE)
        #include <immintrin.h>
    #elif defined(SIMD_SSE_AVAILABLE)
        #include <xmmintrin.h>
        #include <emmintrin.h>
        #ifdef __SSE4_1__
            #include <smmintrin.h>
        #endif
    #endif
#elif defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #ifndef SIMD_NEON_AVAILABLE
        #define SIMD_NEON_AVAILABLE
    #endif
    #undef SIMD_WIDTH
    #define SIMD_WIDTH 4
#else
    #undef SIMD_WIDTH
    #define SIMD_WIDTH 1
#endif

namespace simd_utils {

// ===========================================================================
// SIMD detection
// ===========================================================================

SIMDType get_available_simd()
{
#ifdef SIMD_AVX512_AVAILABLE
    return SIMDType::AVX512;
#elif defined(SIMD_AVX_AVAILABLE)
    return SIMDType::AVX;
#elif defined(SIMD_SSE_AVAILABLE)
    return SIMDType::SSE;
#elif defined(SIMD_NEON_AVAILABLE)
    return SIMDType::NEON;
#else
    return SIMDType::NONE;
#endif
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

namespace scalar
{

void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i) { result[i] = a[i] + b[i]; }
}

void subtract_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i) { result[i] = a[i] - b[i]; }
}

void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i) { result[i] = a[i] * b[i]; }
}

void fma_vectors(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i) { result[i] = a[i] * b[i] + c[i]; }
}

void scale_vectors(float scale, const float* __restrict__ a, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i) { result[i] = scale * a[i]; }
}

void scale_add_vectors(float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i) { result[i] = b[i] + scale * a[i]; }
}

void clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count)
{
    for (int i = 0; i < count; ++i)
    {
        float v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        result[i] = v;
    }
}

int sanitize_nonfinite(const float* __restrict__ a, float fallback, float* __restrict__ result, int count)
{
    int replaced = 0;
    for (int i = 0; i < count; ++i)
    {
        if (std::isfinite(a[i]))
        {
            result[i] = a[i];
        }
        else
        {
            result[i] = fallback;
            ++replaced;
        }
    }
    return replaced;
}

float reduce_sum(const float* __restrict__ a, int count)
{
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) { sum += a[i]; }
    return sum;
}

void reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max)
{
    if (count <= 0)
    {
        *out_min = 0.0f;
        *out_max = 0.0f;
        return;
    }
    float mn = a[0];
    float mx = a[0];
    for (int i = 1; i < count; ++i)
    {
        if (a[i] < mn) mn = a[i];
        if (a[i] > mx) mx = a[i];
    }
    *out_min = mn;
    *out_max = mx;
}

} // namespace scalar

// ===========================================================================
// ISA-specific implementations
// ===========================================================================

#ifdef SIMD_AVX_AVAILABLE

// ---------------------------------------------------------------------------
// AVX path (8 floats / iteration)
// ---------------------------------------------------------------------------

void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    for (; i < simd_end; i += 8)
    {
        _mm256_storeu_ps(&result[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] + b[i]; }
}

void subtract_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    for (; i < simd_end; i += 8)
    {
        _mm256_storeu_ps(&result[i], _mm256_sub_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] - b[i]; }
}

void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    for (; i < simd_end; i += 8)
    {
        _mm256_storeu_ps(&result[i], _mm256_mul_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] * b[i]; }
}

void fma_vectors(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
#ifdef __FMA__
    for (; i < simd_end; i += 8)
    {
        _mm256_storeu_ps(&result[i],
            _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), _mm256_loadu_ps(&c[i])));
    }
#else
    for (; i < simd_end; i += 8)
    {
        __m256 vmul = _mm256_mul_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
        _mm256_storeu_ps(&result[i], _mm256_add_ps(vmul, _mm256_loadu_ps(&c[i])));
    }
#endif
    for (; i < count; ++i) { result[i] = a[i] * b[i] + c[i]; }
}

void scale_vectors(float scale, const float* __restrict__ a, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    const __m256 vs = _mm256_set1_ps(scale);
    for (; i < simd_end; i += 8)
    {
        _mm256_storeu_ps(&result[i], _mm256_mul_ps(vs, _mm256_loadu_ps(&a[i])));
    }
    for (; i < count; ++i) { result[i] = scale * a[i]; }
}

void scale_add_vectors(float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    const __m256 vs = _mm256_set1_ps(scale);
#ifdef __FMA__
    for (; i < simd_end; i += 8)
    {
        _mm256_storeu_ps(&result[i],
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    }
#else
    for (; i < simd_end; i += 8)
    {
        __m256 vmul = _mm256_mul_ps(vs, _mm256_loadu_ps(&a[i]));
        _mm256_storeu_ps(&result[i], _mm256_add_ps(vmul, _mm256_loadu_ps(&b[i])));
    }
#endif
    for (; i < count; ++i) { result[i] = b[i] + scale * a[i]; }
}

void clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vhi = _mm256_set1_ps(hi);
    for (; i < simd_end; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&a[i]);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(&result[i], v);
    }
    for (; i < count; ++i)
    {
        float v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        result[i] = v;
    }
}

int sanitize_nonfinite(const float* __restrict__ a, float fallback, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    const __m256 vfallback = _mm256_set1_ps(fallback);
    int replaced = 0;

    for (; i < simd_end; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&a[i]);
        // ordered compare: v == v is true for finite/normal, false for NaN
        // Then check for inf: abs(v) != inf
        // Combined: cmpord detects NaN; we also need to catch inf.
        // cmpord(v,v) is 0xFFFFFFFF for non-NaN, 0 for NaN
        __m256 not_nan = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
        __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
        __m256 not_inf = _mm256_cmp_ps(abs_v, _mm256_set1_ps(std::numeric_limits<float>::infinity()), _CMP_NEQ_UQ);
        __m256 is_finite = _mm256_and_ps(not_nan, not_inf);
        // Count non-finite: 8 - popcount(movemask(is_finite))
        int finite_mask = _mm256_movemask_ps(is_finite);
        replaced += 8 - __builtin_popcount(finite_mask);
        // Blend: pick v where finite, fallback where not
        _mm256_storeu_ps(&result[i], _mm256_blendv_ps(vfallback, v, is_finite));
    }

    for (; i < count; ++i)
    {
        if (std::isfinite(a[i]))
        {
            result[i] = a[i];
        }
        else
        {
            result[i] = fallback;
            ++replaced;
        }
    }
    return replaced;
}

float reduce_sum(const float* __restrict__ a, int count)
{
    int i = 0;
    const int simd_end = (count / 8) * 8;
    __m256 vsum = _mm256_setzero_ps();
    for (; i < simd_end; i += 8)
    {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&a[i]));
    }
    // Horizontal sum: 8 lanes → 1 float
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    float result;
    _mm_store_ss(&result, sum128);
    for (; i < count; ++i) { result += a[i]; }
    return result;
}

void reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max)
{
    if (count <= 0)
    {
        *out_min = 0.0f;
        *out_max = 0.0f;
        return;
    }
    int i = 0;
    const int simd_end = (count / 8) * 8;
    if (simd_end > 0)
    {
        __m256 vmin = _mm256_loadu_ps(&a[0]);
        __m256 vmax = vmin;
        for (i = 8; i < simd_end; i += 8)
        {
            __m256 v = _mm256_loadu_ps(&a[i]);
            vmin = _mm256_min_ps(vmin, v);
            vmax = _mm256_max_ps(vmax, v);
        }
        // Horizontal reduce 8 → 1
        __m128 lo_min = _mm256_castps256_ps128(vmin);
        __m128 hi_min = _mm256_extractf128_ps(vmin, 1);
        __m128 mn = _mm_min_ps(lo_min, hi_min);
        mn = _mm_min_ps(mn, _mm_movehl_ps(mn, mn));
        mn = _mm_min_ss(mn, _mm_shuffle_ps(mn, mn, 1));
        _mm_store_ss(out_min, mn);

        __m128 lo_max = _mm256_castps256_ps128(vmax);
        __m128 hi_max = _mm256_extractf128_ps(vmax, 1);
        __m128 mx = _mm_max_ps(lo_max, hi_max);
        mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
        mx = _mm_max_ss(mx, _mm_shuffle_ps(mx, mx, 1));
        _mm_store_ss(out_max, mx);
    }
    else
    {
        *out_min = a[0];
        *out_max = a[0];
        i = 1;
    }
    // Scalar tail
    for (; i < count; ++i)
    {
        if (a[i] < *out_min) *out_min = a[i];
        if (a[i] > *out_max) *out_max = a[i];
    }
}

#elif defined(SIMD_SSE_AVAILABLE)

// ---------------------------------------------------------------------------
// SSE path (4 floats / iteration)
// ---------------------------------------------------------------------------

void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        _mm_storeu_ps(&result[i], _mm_add_ps(_mm_loadu_ps(&a[i]), _mm_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] + b[i]; }
}

void subtract_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        _mm_storeu_ps(&result[i], _mm_sub_ps(_mm_loadu_ps(&a[i]), _mm_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] - b[i]; }
}

void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        _mm_storeu_ps(&result[i], _mm_mul_ps(_mm_loadu_ps(&a[i]), _mm_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] * b[i]; }
}

void fma_vectors(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        __m128 vmul = _mm_mul_ps(_mm_loadu_ps(&a[i]), _mm_loadu_ps(&b[i]));
        _mm_storeu_ps(&result[i], _mm_add_ps(vmul, _mm_loadu_ps(&c[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] * b[i] + c[i]; }
}

void scale_vectors(float scale, const float* __restrict__ a, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const __m128 vs = _mm_set1_ps(scale);
    for (; i < simd_end; i += 4)
    {
        _mm_storeu_ps(&result[i], _mm_mul_ps(vs, _mm_loadu_ps(&a[i])));
    }
    for (; i < count; ++i) { result[i] = scale * a[i]; }
}

void scale_add_vectors(float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const __m128 vs = _mm_set1_ps(scale);
    for (; i < simd_end; i += 4)
    {
        __m128 vmul = _mm_mul_ps(vs, _mm_loadu_ps(&a[i]));
        _mm_storeu_ps(&result[i], _mm_add_ps(vmul, _mm_loadu_ps(&b[i])));
    }
    for (; i < count; ++i) { result[i] = b[i] + scale * a[i]; }
}

void clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const __m128 vlo = _mm_set1_ps(lo);
    const __m128 vhi = _mm_set1_ps(hi);
    for (; i < simd_end; i += 4)
    {
        __m128 v = _mm_loadu_ps(&a[i]);
        v = _mm_max_ps(v, vlo);
        v = _mm_min_ps(v, vhi);
        _mm_storeu_ps(&result[i], v);
    }
    for (; i < count; ++i)
    {
        float v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        result[i] = v;
    }
}

int sanitize_nonfinite(const float* __restrict__ a, float fallback, float* __restrict__ result, int count)
{
    // SSE lacks convenient blendv before SSE4.1; use scalar for simplicity.
    return scalar::sanitize_nonfinite(a, fallback, result, count);
}

float reduce_sum(const float* __restrict__ a, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    __m128 vsum = _mm_setzero_ps();
    for (; i < simd_end; i += 4)
    {
        vsum = _mm_add_ps(vsum, _mm_loadu_ps(&a[i]));
    }
    // Horizontal sum: 4 lanes → 1
    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_shuffle_ps(vsum, vsum, 1));
    float result_val;
    _mm_store_ss(&result_val, vsum);
    for (; i < count; ++i) { result_val += a[i]; }
    return result_val;
}

void reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max)
{
    if (count <= 0)
    {
        *out_min = 0.0f;
        *out_max = 0.0f;
        return;
    }
    int i = 0;
    const int simd_end = (count / 4) * 4;
    if (simd_end > 0)
    {
        __m128 vmin = _mm_loadu_ps(&a[0]);
        __m128 vmax = vmin;
        for (i = 4; i < simd_end; i += 4)
        {
            __m128 v = _mm_loadu_ps(&a[i]);
            vmin = _mm_min_ps(vmin, v);
            vmax = _mm_max_ps(vmax, v);
        }
        // Horizontal reduce 4 → 1
        __m128 mn = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
        mn = _mm_min_ss(mn, _mm_shuffle_ps(mn, mn, 1));
        _mm_store_ss(out_min, mn);

        __m128 mx = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
        mx = _mm_max_ss(mx, _mm_shuffle_ps(mx, mx, 1));
        _mm_store_ss(out_max, mx);
    }
    else
    {
        *out_min = a[0];
        *out_max = a[0];
        i = 1;
    }
    for (; i < count; ++i)
    {
        if (a[i] < *out_min) *out_min = a[i];
        if (a[i] > *out_max) *out_max = a[i];
    }
}

#elif defined(SIMD_NEON_AVAILABLE)

// ---------------------------------------------------------------------------
// NEON path (4 floats / iteration)
// ---------------------------------------------------------------------------

void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        vst1q_f32(&result[i], vaddq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] + b[i]; }
}

void subtract_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        vst1q_f32(&result[i], vsubq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] - b[i]; }
}

void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        vst1q_f32(&result[i], vmulq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] * b[i]; }
}

void fma_vectors(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        vst1q_f32(&result[i], vfmaq_f32(vld1q_f32(&c[i]), vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < count; ++i) { result[i] = a[i] * b[i] + c[i]; }
}

void scale_vectors(float scale, const float* __restrict__ a, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const float32x4_t vs = vdupq_n_f32(scale);
    for (; i < simd_end; i += 4)
    {
        vst1q_f32(&result[i], vmulq_f32(vs, vld1q_f32(&a[i])));
    }
    for (; i < count; ++i) { result[i] = scale * a[i]; }
}

void scale_add_vectors(float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const float32x4_t vs = vdupq_n_f32(scale);
    for (; i < simd_end; i += 4)
    {
        // result = b + scale * a  →  vfmaq_f32(b, scale_vec, a)
        vst1q_f32(&result[i], vfmaq_f32(vld1q_f32(&b[i]), vs, vld1q_f32(&a[i])));
    }
    for (; i < count; ++i) { result[i] = b[i] + scale * a[i]; }
}

void clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const float32x4_t vlo = vdupq_n_f32(lo);
    const float32x4_t vhi = vdupq_n_f32(hi);
    for (; i < simd_end; i += 4)
    {
        float32x4_t v = vld1q_f32(&a[i]);
        v = vmaxq_f32(v, vlo);
        v = vminq_f32(v, vhi);
        vst1q_f32(&result[i], v);
    }
    for (; i < count; ++i)
    {
        float v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        result[i] = v;
    }
}

int sanitize_nonfinite(const float* __restrict__ a, float fallback, float* __restrict__ result, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    const float32x4_t vfb = vdupq_n_f32(fallback);
    int replaced = 0;

    for (; i < simd_end; i += 4)
    {
        float32x4_t v = vld1q_f32(&a[i]);
        // A value is finite iff abs(v) < inf. On NEON:
        // vcaltq_f32(v, inf) gives 0xFFFFFFFF for |v| < inf, 0 otherwise.
        // NaN also fails this comparison (NaN < inf is false).
        float32x4_t vinf = vdupq_n_f32(std::numeric_limits<float>::infinity());
        uint32x4_t is_finite = vcaltq_f32(v, vinf);
        // Count non-finite lanes
        // Reinterpret mask as shift-and-add to count set bits
        uint32_t mask[4];
        vst1q_u32(mask, is_finite);
        for (int lane = 0; lane < 4; ++lane)
        {
            if (mask[lane] == 0) ++replaced;
        }
        // Blend: select v where finite, fallback where not
        vst1q_f32(&result[i], vbslq_f32(is_finite, v, vfb));
    }

    for (; i < count; ++i)
    {
        if (std::isfinite(a[i]))
        {
            result[i] = a[i];
        }
        else
        {
            result[i] = fallback;
            ++replaced;
        }
    }
    return replaced;
}

float reduce_sum(const float* __restrict__ a, int count)
{
    int i = 0;
    const int simd_end = (count / 4) * 4;
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i < simd_end; i += 4)
    {
        vsum = vaddq_f32(vsum, vld1q_f32(&a[i]));
    }
    // Horizontal sum: vaddvq_f32 is available on aarch64
    float result_val = vaddvq_f32(vsum);
    for (; i < count; ++i) { result_val += a[i]; }
    return result_val;
}

void reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max)
{
    if (count <= 0)
    {
        *out_min = 0.0f;
        *out_max = 0.0f;
        return;
    }
    int i = 0;
    const int simd_end = (count / 4) * 4;
    if (simd_end > 0)
    {
        float32x4_t vmin = vld1q_f32(&a[0]);
        float32x4_t vmax = vmin;
        for (i = 4; i < simd_end; i += 4)
        {
            float32x4_t v = vld1q_f32(&a[i]);
            vmin = vminq_f32(vmin, v);
            vmax = vmaxq_f32(vmax, v);
        }
        // Horizontal reduce: vminvq_f32 / vmaxvq_f32 available on aarch64
        *out_min = vminvq_f32(vmin);
        *out_max = vmaxvq_f32(vmax);
    }
    else
    {
        *out_min = a[0];
        *out_max = a[0];
        i = 1;
    }
    for (; i < count; ++i)
    {
        if (a[i] < *out_min) *out_min = a[i];
        if (a[i] > *out_max) *out_max = a[i];
    }
}

#else

// ---------------------------------------------------------------------------
// Scalar fallback path (no SIMD available)
// ---------------------------------------------------------------------------

void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count) { scalar::add_vectors(a, b, result, count); }
void subtract_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count) { scalar::subtract_vectors(a, b, result, count); }
void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count) { scalar::multiply_vectors(a, b, result, count); }
void fma_vectors(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ result, int count) { scalar::fma_vectors(a, b, c, result, count); }
void scale_vectors(float scale, const float* __restrict__ a, float* __restrict__ result, int count) { scalar::scale_vectors(scale, a, result, count); }
void scale_add_vectors(float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count) { scalar::scale_add_vectors(scale, a, b, result, count); }
void clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count) { scalar::clamp_vectors(a, lo, hi, result, count); }
int sanitize_nonfinite(const float* __restrict__ a, float fallback, float* __restrict__ result, int count) { return scalar::sanitize_nonfinite(a, fallback, result, count); }
float reduce_sum(const float* __restrict__ a, int count) { return scalar::reduce_sum(a, count); }
void reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max) { scalar::reduce_min_max(a, count, out_min, out_max); }

#endif

// ===========================================================================
// Stencil kernels (shared across all ISA paths)
// ===========================================================================

void diffuse_1d(const float* __restrict__ src, float* __restrict__ dst, float scale, int count)
{
    int i = 0;

#if defined(SIMD_NEON_AVAILABLE)
    const float32x4_t vscale = vdupq_n_f32(scale);
    const float32x4_t vtwo = vdupq_n_f32(2.0f);
    const int simd_end = (count / 4) * 4;

    for (; i < simd_end; i += 4)
    {
        float32x4_t vm = vld1q_f32(&src[i - 1]);
        float32x4_t vc = vld1q_f32(&src[i]);
        float32x4_t vp = vld1q_f32(&src[i + 1]);
        float32x4_t lap = vsubq_f32(vaddq_f32(vm, vp), vmulq_f32(vtwo, vc));
        vst1q_f32(&dst[i], vfmaq_f32(vc, vscale, lap));
    }
#elif defined(SIMD_AVX_AVAILABLE)
    const __m256 vscale = _mm256_set1_ps(scale);
    const __m256 vtwo = _mm256_set1_ps(2.0f);
    const int simd_end = (count / 8) * 8;

    for (; i < simd_end; i += 8)
    {
        __m256 vm = _mm256_loadu_ps(&src[i - 1]);
        __m256 vc = _mm256_loadu_ps(&src[i]);
        __m256 vp = _mm256_loadu_ps(&src[i + 1]);
        __m256 lap = _mm256_sub_ps(_mm256_add_ps(vm, vp), _mm256_mul_ps(vtwo, vc));
    #ifdef __FMA__
        _mm256_storeu_ps(&dst[i], _mm256_fmadd_ps(vscale, lap, vc));
    #else
        _mm256_storeu_ps(&dst[i], _mm256_add_ps(vc, _mm256_mul_ps(vscale, lap)));
    #endif
    }
#elif defined(SIMD_SSE_AVAILABLE)
    const __m128 vscale = _mm_set1_ps(scale);
    const __m128 vtwo = _mm_set1_ps(2.0f);
    const int simd_end = (count / 4) * 4;

    for (; i < simd_end; i += 4)
    {
        __m128 vm = _mm_loadu_ps(&src[i - 1]);
        __m128 vc = _mm_loadu_ps(&src[i]);
        __m128 vp = _mm_loadu_ps(&src[i + 1]);
        __m128 lap = _mm_sub_ps(_mm_add_ps(vm, vp), _mm_mul_ps(vtwo, vc));
        _mm_storeu_ps(&dst[i], _mm_add_ps(vc, _mm_mul_ps(vscale, lap)));
    }
#else
    (void)i;
    const int simd_end = 0;
#endif

    for (; i < count; ++i)
    {
        float lap = src[i - 1] - 2.0f * src[i] + src[i + 1];
        dst[i] = src[i] + scale * lap;
    }
}

} // namespace simd_utils
