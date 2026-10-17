#pragma once

#include <algorithm>
#include <array>
#include <string_view>

namespace tmv::runtime
{
inline constexpr std::string_view kGeneratedMetalOutputScopeV2ManifestSha256 = "41144394ad21ee2df018d319061607c3a6423d617aeb94a7fa27053505491e4a";
inline constexpr int kGeneratedMetalOutputScopeV2TupleCount = 6048;
inline constexpr int kGeneratedMetalOutputScopeV2DistinctConfigCount = 4536;
inline constexpr std::array<int, 2> kGeneratedMetalNpyWriteCadenceSeconds{1, 301};
inline constexpr std::array<int, 2> kGeneratedMetalNetcdfWriteCadenceSeconds{1, 301};
inline constexpr std::array<int, 1> kGeneratedMetalZfpHighWriteCadenceSeconds{1};

inline bool admits_generated_metal_output_write_cadence(std::string_view format, int seconds) noexcept
{
    if (format == "npy_3d")
    {
        return std::binary_search(kGeneratedMetalNpyWriteCadenceSeconds.begin(),
                                  kGeneratedMetalNpyWriteCadenceSeconds.end(), seconds);
    }
    if (format == "netcdf")
    {
        return std::binary_search(kGeneratedMetalNetcdfWriteCadenceSeconds.begin(),
                                  kGeneratedMetalNetcdfWriteCadenceSeconds.end(), seconds);
    }
    if (format == "zfp_high")
    {
        return std::binary_search(kGeneratedMetalZfpHighWriteCadenceSeconds.begin(),
                                  kGeneratedMetalZfpHighWriteCadenceSeconds.end(), seconds);
    }
    return false;
}
} // namespace tmv::runtime
