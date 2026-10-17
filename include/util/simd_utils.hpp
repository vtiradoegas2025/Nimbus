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
// There is no native 512-bit implementation below yet.  AVX-512 targets also
// execute the established 256-bit AVX kernels instead of falling through to
// scalar code, and the advertised width must describe that active path.
#define SIMD_AVX_AVAILABLE
#define SIMD_WIDTH 8
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
    NONE   = 0,
    SSE    = 1,
    AVX    = 2,
    AVX512 = 3,
    NEON   = 4
};

struct SIMDFeatures
{
    bool avx512 = false;
    bool avx    = false;
    bool sse    = false;
    bool neon   = false;
};

/** Selects the implementation used by this utility layer.
 *
 * AVX-512 compilation intentionally maps to the tested AVX-width kernels until
 * a distinct 512-bit implementation exists.  Keeping this policy independent
 * of host preprocessor macros makes forced-path tests portable.
 */
constexpr SIMDType
select_simd_implementation(bool avx512_compiled, bool avx_compiled, bool sse_compiled, bool neon_compiled)
{
    return (avx512_compiled || avx_compiled) ? SIMDType::AVX
           : sse_compiled                    ? SIMDType::SSE
           : neon_compiled                   ? SIMDType::NEON
                                             : SIMDType::NONE;
}

/**
 * Select the best implementation present in the binary and supported by the
 * executing host. AVX-512 variants currently map to the tested AVX-width path.
 */
constexpr SIMDType select_runtime_simd(SIMDFeatures compiled, SIMDFeatures runtime)
{
    return ((compiled.avx512 || compiled.avx) && runtime.avx) ? SIMDType::AVX
           : (compiled.sse && runtime.sse)                    ? SIMDType::SSE
           : (compiled.neon && runtime.neon)                  ? SIMDType::NEON
                                                              : SIMDType::NONE;
}

/** Number of float lanes processed by the selected implementation. */
constexpr int simd_implementation_width(SIMDType type)
{
    return type == SIMDType::AVX512                            ? 16
           : type == SIMDType::AVX                             ? 8
           : (type == SIMDType::SSE || type == SIMDType::NEON) ? 4
                                                               : 1;
}

/**
 * @brief Reports the SIMD implementation active in this native build.
 * @return Selected implementation type, not merely the compiler's maximum ISA.
 */
SIMDType get_available_simd();

/** Host capabilities independent of which implementations were compiled. */
SIMDFeatures get_runtime_simd_features();

/** Implementations physically present in this binary. */
SIMDFeatures get_compiled_simd_features();

// ---------------------------------------------------------------------------
// Element-wise arithmetic
// ---------------------------------------------------------------------------

/**
 * @brief Computes element-wise vector addition: result[i] = a[i] + b[i].
 */
void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count);

/**
 * @brief Computes element-wise vector subtraction: result[i] = a[i] - b[i].
 *
 * Exact in-place aliasing is supported: result may equal a, b, or both. The
 * ranges must otherwise be identical or non-overlapping (no shifted overlap).
 */
void subtract_vectors(const float* a, const float* b, float* result, int count);

/**
 * @brief Computes element-wise vector multiplication: result[i] = a[i] * b[i].
 */
void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count);

/**
 * @brief Computes element-wise fused multiply-add: result[i] = a[i] * b[i] + c[i].
 */
void fma_vectors(const float* __restrict__ a,
                 const float* __restrict__ b,
                 const float* __restrict__ c,
                 float* __restrict__ result,
                 int count);

// ---------------------------------------------------------------------------
// Scalar-broadcast operations
// ---------------------------------------------------------------------------

/**
 * @brief Scales every element: result[i] = scale * a[i].
 *
 * Exact in-place aliasing (result == a) is supported.
 */
void scale_vectors(float scale, const float* a, float* result, int count);

/**
 * @brief Scalar-broadcast fused multiply-add: result[i] = b[i] + scale * a[i].
 *
 * The most common pattern in tendency updates: field += dt * tendency.
 */
void scale_add_vectors(
    float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count);

// ---------------------------------------------------------------------------
// Clamping and sanitization
// ---------------------------------------------------------------------------

/**
 * @brief Clamps every element to [lo, hi]: result[i] = clamp(a[i], lo, hi).
 *
 * Uses branchless min/max intrinsics on all SIMD paths.
 */
void clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count);

/**
 * @brief Replaces non-finite values with a fallback: result[i] = isfinite(a[i]) ? a[i] : fallback.
 * @return Number of non-finite elements that were replaced.
 *
 * Uses branchless NaN/Inf detection via ordered-comparison intrinsics.
 * Exact in-place aliasing (result == a) is supported.
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
float reduce_sum(const float* __restrict__ a, int count);

/**
 * @brief Computes the minimum and maximum of all elements in a single pass.
 * @param[out] out_min Receives the minimum value.
 * @param[out] out_max Receives the maximum value.
 *
 * For count <= 0, out_min and out_max are set to 0.0f.
 */
void reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max);

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
void diffuse_1d(const float* __restrict__ src, float* __restrict__ dst, float scale, int count);

// ---------------------------------------------------------------------------
// Scalar fallbacks (testing and verification)
// ---------------------------------------------------------------------------

namespace scalar
{
void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count);
void subtract_vectors(const float* a, const float* b, float* result, int count);
void multiply_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count);
void fma_vectors(const float* __restrict__ a,
                 const float* __restrict__ b,
                 const float* __restrict__ c,
                 float* __restrict__ result,
                 int count);
void scale_vectors(float scale, const float* a, float* result, int count);
void scale_add_vectors(
    float scale, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int count);
void  clamp_vectors(const float* __restrict__ a, float lo, float hi, float* __restrict__ result, int count);
int   sanitize_nonfinite(const float* a, float fallback, float* result, int count);
float reduce_sum(const float* __restrict__ a, int count);
void  reduce_min_max(const float* __restrict__ a, int count, float* __restrict__ out_min, float* __restrict__ out_max);
} // namespace scalar

} // namespace simd_utils
