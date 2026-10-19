#include "core/infra/sha256.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

#if !defined(NIMBUS_SHA256_PORTABLE_ONLY) && defined(__ARM_FEATURE_SHA2)
#include <arm_neon.h>
#elif !defined(NIMBUS_SHA256_PORTABLE_ONLY) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace nimbus::crypto
{
namespace
{

constexpr std::array<std::uint32_t, 64> k = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned bits) noexcept
{
    return (value >> bits) | (value << (32U - bits));
}

using Compression = void (*)(std::uint32_t*, const std::uint32_t*) noexcept;

void compress_portable(std::uint32_t* state, const std::uint32_t* words) noexcept
{
    auto a = state[0], b = state[1], c = state[2], d = state[3];
    auto e = state[4], f = state[5], g = state[6], h = state[7];
    for (std::size_t index = 0; index < 64; ++index)
    {
        const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temporary1 = h + sum1 + choose + k[index] + words[index];
        const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

#if !defined(NIMBUS_SHA256_PORTABLE_ONLY) && defined(__ARM_FEATURE_SHA2)
void compress_arm(std::uint32_t* state, const std::uint32_t* words) noexcept
{
    auto abcd = vld1q_u32(state);
    auto efgh = vld1q_u32(state + 4);
    for (std::size_t round = 0; round < 64; round += 4)
    {
        const auto message = vaddq_u32(vld1q_u32(words + round), vld1q_u32(k.data() + round));
        const auto previous_abcd = abcd;
        abcd = vsha256hq_u32(abcd, efgh, message);
        efgh = vsha256h2q_u32(efgh, previous_abcd, message);
    }
    vst1q_u32(state, vaddq_u32(vld1q_u32(state), abcd));
    vst1q_u32(state + 4, vaddq_u32(vld1q_u32(state + 4), efgh));
}
#elif !defined(NIMBUS_SHA256_PORTABLE_ONLY) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sha,sse2")))
void compress_x86(std::uint32_t* state, const std::uint32_t* words) noexcept
{
    // SHA256RNDS2 uses ABEF/CDGH in high-to-low lane order, unlike the stored ABCDEFGH state.
    std::uint32_t abef[4] = {state[5], state[4], state[1], state[0]};
    std::uint32_t cdgh[4] = {state[7], state[6], state[3], state[2]};
    auto first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(abef));
    auto second = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cdgh));
    for (std::size_t round = 0; round < 64; round += 4)
    {
        const auto message = _mm_add_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(words + round)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(k.data() + round)));
        second = _mm_sha256rnds2_epu32(second, first, message);
        first = _mm_sha256rnds2_epu32(first, second, _mm_srli_si128(message, 8));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(abef), first);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(cdgh), second);
    state[0] += abef[3]; state[1] += abef[2];
    state[2] += cdgh[3]; state[3] += cdgh[2];
    state[4] += abef[1]; state[5] += abef[0];
    state[6] += cdgh[1]; state[7] += cdgh[0];
}
#endif

Compression compression() noexcept
{
#if !defined(NIMBUS_SHA256_PORTABLE_ONLY) && defined(__ARM_FEATURE_SHA2)
    return compress_arm;
#elif !defined(NIMBUS_SHA256_PORTABLE_ONLY) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool available = [] {
        unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
        return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1U << 29));
    }();
    return available ? compress_x86 : compress_portable;
#else
    return compress_portable;
#endif
}

} // namespace

const char* sha256_implementation() noexcept
{
    return compression() == compress_portable ? "portable" :
#if defined(__ARM_FEATURE_SHA2)
        "arm_sha2";
#else
        "x86_sha";
#endif
}

Sha256::Sha256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}
{
}

void Sha256::transform(const std::uint8_t block[64]) noexcept
{
    std::uint32_t words[64]{};
    for (std::size_t index = 0; index < 16; ++index)
    {
        const std::size_t offset = index * 4;
        words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                       (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                       (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                       static_cast<std::uint32_t>(block[offset + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index)
    {
        const std::uint32_t s0 = rotate_right(words[index - 15], 7) ^
                                 rotate_right(words[index - 15], 18) ^
                                 (words[index - 15] >> 3U);
        const std::uint32_t s1 = rotate_right(words[index - 2], 17) ^
                                 rotate_right(words[index - 2], 19) ^
                                 (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    compression()(state_.data(), words);
}

void Sha256::update(const void* data, std::size_t size) noexcept
{
    if (size == 0) return;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    byte_count_ += static_cast<std::uint64_t>(size);
    if (buffer_size_ != 0)
    {
        const auto count = std::min(size, buffer_.size() - buffer_size_);
        std::memcpy(buffer_.data() + buffer_size_, bytes, count);
        buffer_size_ += count;
        bytes += count;
        size -= count;
        if (buffer_size_ == buffer_.size())
        {
            transform(buffer_.data());
            buffer_size_ = 0;
        }
    }
    while (size >= buffer_.size())
    {
        transform(bytes);
        bytes += buffer_.size();
        size -= buffer_.size();
    }
    if (size != 0)
    {
        std::memcpy(buffer_.data(), bytes, size);
        buffer_size_ = size;
    }
}

std::array<std::uint8_t, 32> Sha256::digest() const noexcept
{
    Sha256 finalized = *this;
    const std::uint64_t bit_count = finalized.byte_count_ * 8U;
    const std::uint8_t marker = 0x80U;
    finalized.update(&marker, 1);
    const std::uint8_t zero = 0;
    while (finalized.buffer_size_ != 56)
        finalized.update(&zero, 1);
    std::uint8_t length[8]{};
    for (std::size_t index = 0; index < 8; ++index)
        length[7 - index] = static_cast<std::uint8_t>(bit_count >> (index * 8U));
    finalized.update(length, sizeof(length));

    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < finalized.state_.size(); ++index)
    {
        result[index * 4] = static_cast<std::uint8_t>(finalized.state_[index] >> 24U);
        result[index * 4 + 1] = static_cast<std::uint8_t>(finalized.state_[index] >> 16U);
        result[index * 4 + 2] = static_cast<std::uint8_t>(finalized.state_[index] >> 8U);
        result[index * 4 + 3] = static_cast<std::uint8_t>(finalized.state_[index]);
    }
    return result;
}

std::string Sha256::hex_digest() const
{
    const auto bytes = digest();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint8_t byte : bytes)
        out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::string sha256_hex(std::string_view value)
{
    Sha256 hash;
    hash.update(value.data(), value.size());
    return hash.hex_digest();
}

} // namespace nimbus::crypto
