#pragma once

#include <cstddef>
#include <cstdint>

namespace tmv::soundings
{

inline constexpr std::uintmax_t kMaxTextSoundingFileBytes = 8u * 1024u * 1024u;
inline constexpr std::size_t kMaxTextSoundingLevels = 100000;
inline constexpr std::size_t kMaxTextSoundingLineBytes = 4096;
inline constexpr std::size_t kMaxSoundingMetadataBytes = 1023;

} // namespace tmv::soundings
