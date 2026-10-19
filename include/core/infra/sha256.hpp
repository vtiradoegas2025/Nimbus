#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace nimbus::crypto
{

const char* sha256_implementation() noexcept;

class Sha256
{
public:
    Sha256();

    void update(const void* data, std::size_t size) noexcept;
    std::array<std::uint8_t, 32> digest() const noexcept;
    std::string hex_digest() const;

private:
    void transform(const std::uint8_t block[64]) noexcept;

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t byte_count_ = 0;
    std::size_t buffer_size_ = 0;
};

std::string sha256_hex(std::string_view value);

} // namespace nimbus::crypto
