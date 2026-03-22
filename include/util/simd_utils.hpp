#pragma once

#include <cstdint>
#include <cstring>

/**
 * @file simd_utils.hpp
 * @brief SIMD capability detection and vector math utility declarations.
 *
 * Declares architecture-aware vector kernels and scalar fallbacks.
 * Compile-time macros expose the available SIMD width for callers.
 * Implementations choose optimized instructions when available.
 */

namespace simd_utils
{

#if defined(__AVX512F__)
#define SIMD_AVX512_AVAILABLE
#define SIMD_WIDTH 16
#elif defined(__AVX__)
#define SIMD_AVX_AVAILABLE
#define SIMD_WIDTH 8
#elif defined(__SSE4_1__) || defined(__SSE__)
#define SIMD_SSE_AVAILABLE
#define SIMD_WIDTH 4
#elif defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_NEON_AVAILABLE
#define SIMD_WIDTH 4
#else
#define SIMD_WIDTH 1
#endif

enum class SIMDType
{
    NONE = 0,
    SSE = 1,
    AVX = 2,
    AVX512 = 3,
    NEON = 4
};

/**
 * @brief Detects the SIMD ISA supported by the current build/runtime path.
 * @return Selected SIMD type.
 */
SIMDType get_available_simd();

// ---------------------------------------------------------------------------
// Element-wise arithmetic
// ---------------------------------------------------------------------------

/**
 * @brief Computes element-wise vector addition: result[i] = a[i] + b[i].
 */
void add_vectors(const float* a, const float* b, float* result, int count);

/**
 * @brief Computes element-wise vector subtraction: result[i] = a[i] - b[i].
 */
void subtract_vectors(const float* a, const float* b, float* result, int count);

/**
 * @brief Computes element-wise vector multiplication: result[i] = a[i] * b[i].
 */
void multiply_vectors(const float* a, const float* b, float* result, int count);

/**
 * @brief Computes element-wise fused multiply-add: result[i] = a[i] * b[i] + c[i].
 */
void fma_vectors(const float* a, const float* b, const float* c, float* result, int count);

// ---------------------------------------------------------------------------
// Scalar-broadcast operations
// ---------------------------------------------------------------------------

/**
 * @brief Scales every element: result[i] = scale * a[i].
 */
void scale_vectors(float scale, const float* a, float* result, int count);

/**
 * @brief Scalar-broadcast fused multiply-add: result[i] = b[i] + scale * a[i].
 *
 * The most common pattern in tendency updates: field += dt * tendency.
 */
void scale_add_vectors(float scale, const float* a, const float* b, float* result, int count);

// ---------------------------------------------------------------------------
// Clamping and sanitization
// ---------------------------------------------------------------------------

/**
 * @brief Clamps every element to [lo, hi]: result[i] = clamp(a[i], lo, hi).
 *
 * Uses branchless min/max intrinsics on all SIMD paths.
 */
void clamp_vectors(const float* a, float lo, float hi, float* result, int count);

/**
 * @brief Replaces non-finite values with a fallback: result[i] = isfinite(a[i]) ? a[i] : fallback.
 * @return Number of non-finite elements that were replaced.
 *
 * Uses branchless NaN/Inf detection via ordered-comparison intrinsics.
 */
int sanitize_nonfinite(const float* a, float fallback, float* result, int count);

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/**
 * @brief Computes the sum of all elements: return Σa[i].
 *
 * Uses SIMD tree-reduction with a horizontal add for the final lane merge.
 */
float reduce_sum(const float* a, int count);

/**
 * @brief Computes the minimum and maximum of all elements in a single pass.
 * @param[out] out_min Receives the minimum value.
 * @param[out] out_max Receives the maximum value.
 *
 * For count <= 0, out_min and out_max are set to 0.0f.
 */
void reduce_min_max(const float* a, int count, float* out_min, float* out_max);

// ---------------------------------------------------------------------------
// Stencil kernels
// ---------------------------------------------------------------------------

/**
 * @brief Computes dst[i] = src[i] + scale * (src[i-1] - 2*src[i] + src[i+1])
 *        for i in [0, count). Caller must ensure src has valid data at [-1] and [count].
 *
 * This is the inner-loop kernel for 1D Laplacian diffusion along a stride-1
 * dimension. On NEON/SSE/AVX, processes 4-8 elements per iteration.
 */
void diffuse_1d(const float* src, float* dst, float scale, int count);

// ---------------------------------------------------------------------------
// Scalar fallbacks (testing and verification)
// ---------------------------------------------------------------------------

namespace scalar
{
void add_vectors(const float* a, const float* b, float* result, int count);
void subtract_vectors(const float* a, const float* b, float* result, int count);
void multiply_vectors(const float* a, const float* b, float* result, int count);
void fma_vectors(const float* a, const float* b, const float* c, float* result, int count);
void scale_vectors(float scale, const float* a, float* result, int count);
void scale_add_vectors(float scale, const float* a, const float* b, float* result, int count);
void clamp_vectors(const float* a, float lo, float hi, float* result, int count);
int sanitize_nonfinite(const float* a, float fallback, float* result, int count);
float reduce_sum(const float* a, int count);
void reduce_min_max(const float* a, int count, float* out_min, float* out_max);
} // namespace scalar

} // namespace simd_utils
